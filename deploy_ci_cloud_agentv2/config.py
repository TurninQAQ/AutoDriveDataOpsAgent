"""Strict, secret-free configuration for the production runtime host."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from .agent.budgets import RuntimeBudgets


class ConfigurationError(ValueError):
    """The host configuration is missing, malformed, or contains unknown keys."""


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    model: str
    endpoint: str
    api_key_env: str
    connect_timeout_seconds: float = 5.0
    read_timeout_seconds: float = 60.0
    overall_timeout_seconds: float = 90.0
    max_retries: int = 2
    retry_backoff_seconds: float = 0.25

    def __post_init__(self) -> None:
        _non_empty(self.name, "provider.name")
        _non_empty(self.model, "provider.model")
        _non_empty(self.api_key_env, "provider.api_key_env")
        _http_endpoint(self.endpoint, "provider.endpoint")
        _positive(self.connect_timeout_seconds, "provider.connect_timeout_seconds")
        _positive(self.read_timeout_seconds, "provider.read_timeout_seconds")
        _positive(self.overall_timeout_seconds, "provider.overall_timeout_seconds")
        _non_negative_int(self.max_retries, "provider.max_retries")
        _non_negative(self.retry_backoff_seconds, "provider.retry_backoff_seconds")


@dataclass(frozen=True)
class PlatformConfig:
    endpoint: str
    api_key_env: str | None = None
    connect_timeout_seconds: float = 5.0
    read_timeout_seconds: float = 30.0
    overall_timeout_seconds: float = 45.0
    max_retries: int = 2
    retry_backoff_seconds: float = 0.1

    def __post_init__(self) -> None:
        _http_endpoint(self.endpoint, "platform.endpoint")
        if self.api_key_env is not None:
            _non_empty(self.api_key_env, "platform.api_key_env")
        _positive(self.connect_timeout_seconds, "platform.connect_timeout_seconds")
        _positive(self.read_timeout_seconds, "platform.read_timeout_seconds")
        _positive(self.overall_timeout_seconds, "platform.overall_timeout_seconds")
        _non_negative_int(self.max_retries, "platform.max_retries")
        _non_negative(self.retry_backoff_seconds, "platform.retry_backoff_seconds")


@dataclass(frozen=True)
class PersistenceConfig:
    runtime_root: Path
    sqlite_path: Path
    single_instance: bool = True

    def __post_init__(self) -> None:
        if type(self.single_instance) is not bool:
            raise ConfigurationError("persistence.single_instance must be a boolean")
        if self.single_instance is not True:
            raise ConfigurationError(
                "persistence.single_instance=false is unsupported; "
                "the current Runtime requires one active instance"
            )
        if not self.runtime_root.is_absolute():
            raise ConfigurationError("persistence.runtime_root must be absolute")
        if not self.sqlite_path.is_absolute():
            raise ConfigurationError("persistence.sqlite_path must be absolute")


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    json_logs: bool = True

    def __post_init__(self) -> None:
        if self.level.upper() not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ConfigurationError("logging.level is invalid")
        if type(self.json_logs) is not bool:
            raise ConfigurationError("logging.json_logs must be a boolean")


@dataclass(frozen=True)
class RuntimeConfig:
    environment: str
    operator_id: str
    trust_domain: str
    provider: ProviderConfig
    platform: PlatformConfig
    persistence: PersistenceConfig
    logging: LoggingConfig
    principles_path: Path
    budgets: RuntimeBudgets

    def __post_init__(self) -> None:
        _non_empty(self.environment, "environment")
        _non_empty(self.operator_id, "operator_id")
        _non_empty(self.trust_domain, "trust_domain")
        if not self.principles_path.is_file():
            raise ConfigurationError(
                f"principles file is unavailable: {self.principles_path}"
            )

    @classmethod
    def from_env(
        cls,
        *,
        runtime_root: str | Path = "/home/ubuntu/project/autodrive_dataops_runtimev2",
        config_path: str | Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> "RuntimeConfig":
        env = dict(os.environ if environ is None else environ)
        root = Path(env.get("AUTODRIVE_RUNTIME_ROOT", runtime_root)).expanduser().resolve()
        file_values = _load_json_config(config_path) if config_path is not None else {}
        _strict_keys(
            file_values,
            {"environment", "operator_id", "trust_domain", "provider", "platform", "persistence", "logging", "principles_path", "budgets"},
            "runtime config",
        )

        provider_values = _section(file_values, "provider")
        platform_values = _section(file_values, "platform")
        persistence_values = _section(file_values, "persistence")
        logging_values = _section(file_values, "logging")
        budget_values = _section(file_values, "budgets")
        _strict_keys(provider_values, {field.name for field in fields(ProviderConfig)}, "provider")
        _strict_keys(platform_values, {field.name for field in fields(PlatformConfig)}, "platform")
        _strict_keys(persistence_values, {field.name for field in fields(PersistenceConfig)}, "persistence")
        _strict_keys(logging_values, {field.name for field in fields(LoggingConfig)}, "logging")
        _strict_keys(budget_values, {field.name for field in fields(RuntimeBudgets)}, "budgets")

        provider = ProviderConfig(
            name=_env_or(env, "AUTODRIVE_PROVIDER", provider_values, "name", "qwen"),
            model=_env_or(env, "AUTODRIVE_MODEL", provider_values, "model", "qwen-plus"),
            endpoint=_env_or(env, "AUTODRIVE_PROVIDER_ENDPOINT", provider_values, "endpoint", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"),
            api_key_env=_env_or(env, "AUTODRIVE_PROVIDER_API_KEY_ENV", provider_values, "api_key_env", "DASHSCOPE_API_KEY"),
            connect_timeout_seconds=_float_env(env, "AUTODRIVE_PROVIDER_CONNECT_TIMEOUT", provider_values, "connect_timeout_seconds", 5.0),
            read_timeout_seconds=_float_env(env, "AUTODRIVE_PROVIDER_READ_TIMEOUT", provider_values, "read_timeout_seconds", 60.0),
            overall_timeout_seconds=_float_env(env, "AUTODRIVE_PROVIDER_OVERALL_TIMEOUT", provider_values, "overall_timeout_seconds", 90.0),
            max_retries=_int_env(env, "AUTODRIVE_PROVIDER_MAX_RETRIES", provider_values, "max_retries", 2),
            retry_backoff_seconds=_float_env(env, "AUTODRIVE_PROVIDER_RETRY_BACKOFF", provider_values, "retry_backoff_seconds", 0.25),
        )
        platform = PlatformConfig(
            endpoint=_env_or(env, "AUTODRIVE_PLATFORM_ENDPOINT", platform_values, "endpoint", "http://127.0.0.1:8765/mcp"),
            api_key_env=_optional_env_or(env, "AUTODRIVE_PLATFORM_API_KEY_ENV", platform_values, "api_key_env"),
            connect_timeout_seconds=_float_env(env, "AUTODRIVE_PLATFORM_CONNECT_TIMEOUT", platform_values, "connect_timeout_seconds", 5.0),
            read_timeout_seconds=_float_env(env, "AUTODRIVE_PLATFORM_READ_TIMEOUT", platform_values, "read_timeout_seconds", 30.0),
            overall_timeout_seconds=_float_env(env, "AUTODRIVE_PLATFORM_OVERALL_TIMEOUT", platform_values, "overall_timeout_seconds", 45.0),
            max_retries=_int_env(env, "AUTODRIVE_PLATFORM_MAX_RETRIES", platform_values, "max_retries", 2),
            retry_backoff_seconds=_float_env(env, "AUTODRIVE_PLATFORM_RETRY_BACKOFF", platform_values, "retry_backoff_seconds", 0.1),
        )
        sqlite_raw = _env_or(
            env,
            "AUTODRIVE_SQLITE_PATH",
            persistence_values,
            "sqlite_path",
            str(root / "state" / "autodrive.sqlite3"),
        )
        runtime_root_raw = _env_or(
            env, "AUTODRIVE_RUNTIME_ROOT", persistence_values, "runtime_root", str(root)
        )
        _non_empty_path(sqlite_raw, "persistence.sqlite_path")
        _non_empty_path(runtime_root_raw, "persistence.runtime_root")
        sqlite_path = Path(sqlite_raw).expanduser().resolve()
        persistence = PersistenceConfig(
            runtime_root=Path(runtime_root_raw).expanduser().resolve(),
            sqlite_path=sqlite_path,
            single_instance=_bool_env(env, "AUTODRIVE_SINGLE_INSTANCE", persistence_values, "single_instance", True),
        )
        principles = Path(
            _env_or(
                env,
                "AUTODRIVE_PRINCIPLES_PATH",
                file_values,
                "principles_path",
                str(Path(__file__).resolve().parent / "doc" / "Luna_OPERATING_PRINCIPLES.md"),
            )
        ).expanduser().resolve()
        budgets = RuntimeBudgets(
            **{
                name: _typed_budget_value(env, name, budget_values, getattr(RuntimeBudgets(), name))
                for name in {field.name for field in fields(RuntimeBudgets)}
            }
        )
        return cls(
            environment=_env_or(env, "AUTODRIVE_ENVIRONMENT", file_values, "environment", "development"),
            operator_id=_env_or(env, "AUTODRIVE_OPERATOR_ID", file_values, "operator_id", "trusted-operator"),
            trust_domain=_env_or(env, "AUTODRIVE_TRUST_DOMAIN", file_values, "trust_domain", "default-trust-domain"),
            provider=provider,
            platform=platform,
            persistence=persistence,
            logging=LoggingConfig(
                level=_env_or(env, "AUTODRIVE_LOG_LEVEL", logging_values, "level", "INFO"),
                json_logs=_bool_env(env, "AUTODRIVE_JSON_LOGS", logging_values, "json_logs", True),
            ),
            principles_path=principles,
            budgets=budgets,
        )


def ensure_runtime_layout(config: RuntimeConfig) -> None:
    """Create only the non-secret runtime directories, outside the source tree."""
    for name in ("config", "data", "state", "logs", "run", "secrets"):
        (config.persistence.runtime_root / name).mkdir(parents=True, exist_ok=True)


def _load_json_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"cannot load config file: {source}") from exc
    if type(value) is not dict:
        raise ConfigurationError("config file root must be a JSON object")
    return value


def _strict_keys(value: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ConfigurationError(f"unknown {name} keys: {', '.join(sorted(unknown))}")


def _section(root: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = root.get(name, {})
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{name} must be an object")
    return dict(value)


def _env_or(env: Mapping[str, str], env_name: str, values: Mapping[str, Any], key: str, default: Any) -> Any:
    return env.get(env_name, values.get(key, default))


def _optional_env_or(env: Mapping[str, str], env_name: str, values: Mapping[str, Any], key: str) -> str | None:
    if env_name in env:
        return env[env_name] or None
    return values.get(key)


def _typed_budget_value(env: Mapping[str, str], name: str, values: Mapping[str, Any], default: Any) -> Any:
    env_name = "AUTODRIVE_" + "_".join(part.upper() for part in name.split("_"))
    raw = env.get(env_name, values.get(name, default))
    if type(default) is not int or isinstance(raw, bool):
        raise ConfigurationError(f"budget {name} must be an integer")
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"budget {name} must be an integer") from exc


def _float_env(env: Mapping[str, str], env_name: str, values: Mapping[str, Any], key: str, default: float) -> float:
    raw = env.get(env_name, values.get(key, default))
    if isinstance(raw, bool):
        raise ConfigurationError(f"{key} must be numeric")
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{key} must be numeric") from exc


def _int_env(env: Mapping[str, str], env_name: str, values: Mapping[str, Any], key: str, default: int) -> int:
    raw = env.get(env_name, values.get(key, default))
    if isinstance(raw, bool):
        raise ConfigurationError(f"{key} must be an integer")
    if not isinstance(raw, (str, int)):
        raise ConfigurationError(f"{key} must be an integer")
    try:
        parsed = int(raw)
        if isinstance(raw, str) and str(parsed) != raw.strip():
            raise ValueError
        return parsed
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{key} must be an integer") from exc


def _bool_env(env: Mapping[str, str], env_name: str, values: Mapping[str, Any], key: str, default: bool) -> bool:
    raw = env.get(env_name, values.get(key, default))
    if type(raw) is bool:
        return raw
    if isinstance(raw, str) and raw.lower() in {"true", "1", "yes"}:
        return True
    if isinstance(raw, str) and raw.lower() in {"false", "0", "no"}:
        return False
    raise ConfigurationError(f"{key} must be boolean")


def _non_empty(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{name} must be non-empty")


def _non_empty_path(value: object, name: str) -> None:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ConfigurationError(f"{name} must be a non-empty path")


def _positive(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ConfigurationError(f"{name} must be positive")


def _non_negative(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ConfigurationError(f"{name} must not be negative")


def _non_negative_int(value: object, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ConfigurationError(f"{name} must be a non-negative integer")


def _http_endpoint(value: object, name: str) -> None:
    _non_empty(value, name)
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ConfigurationError(f"{name} must be an http(s) URL")
