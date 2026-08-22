"""Small strict primitives for external response contracts.

The distinction between a missing field and a malformed field is a Runtime
invariant.  This module deliberately does not coerce values from external
payloads: a value is either absent, exactly valid for its contract, or invalid.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, Mapping, TypeVar


class FieldState(str, Enum):
    ABSENT = "ABSENT"
    PRESENT_VALID = "PRESENT_VALID"
    PRESENT_INVALID = "PRESENT_INVALID"


T = TypeVar("T")


@dataclass(frozen=True)
class FieldResult(Generic[T]):
    state: FieldState
    value: T | None = None
    error: str | None = None

    @property
    def is_absent(self) -> bool:
        return self.state is FieldState.ABSENT

    @property
    def is_valid(self) -> bool:
        return self.state is FieldState.PRESENT_VALID

    @property
    def is_invalid(self) -> bool:
        return self.state is FieldState.PRESENT_INVALID


def _absent() -> FieldResult[Any]:
    return FieldResult(FieldState.ABSENT)


def _invalid(name: str, expected: str, actual: object) -> FieldResult[Any]:
    return FieldResult(
        FieldState.PRESENT_INVALID,
        error=f"{name} must be {expected}; got {type(actual).__name__}",
    )


def read_optional_bool(raw: Mapping[str, Any], name: str, *, nullable: bool = False) -> FieldResult[bool | None]:
    if name not in raw:
        return _absent()
    value = raw[name]
    if isinstance(value, bool):
        return FieldResult(FieldState.PRESENT_VALID, value)
    if nullable and value is None:
        return FieldResult(FieldState.PRESENT_VALID, None)
    return _invalid(name, "a boolean", value)


def read_optional_string(
    raw: Mapping[str, Any],
    name: str,
    *,
    nullable: bool = False,
    non_empty: bool = True,
) -> FieldResult[str | None]:
    if name not in raw:
        return _absent()
    value = raw[name]
    if nullable and value is None:
        return FieldResult(FieldState.PRESENT_VALID, None)
    if not isinstance(value, str):
        return _invalid(name, "a string", value)
    normalized = value.strip()
    if non_empty and not normalized:
        return FieldResult(FieldState.PRESENT_INVALID, error=f"{name} must be non-empty")
    return FieldResult(FieldState.PRESENT_VALID, normalized)


def read_optional_enum(
    raw: Mapping[str, Any],
    name: str,
    allowed: Mapping[str, T],
    *,
    nullable: bool = False,
) -> FieldResult[T | None]:
    field = read_optional_string(raw, name, nullable=nullable)
    if not field.is_valid or field.value is None:
        return field
    value = allowed.get(field.value.upper())
    if value is None:
        return FieldResult(FieldState.PRESENT_INVALID, error=f"{name} has an unknown value")
    return FieldResult(FieldState.PRESENT_VALID, value)


def read_optional_int(
    raw: Mapping[str, Any],
    name: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
    nullable: bool = False,
) -> FieldResult[int | None]:
    if name not in raw:
        return _absent()
    value = raw[name]
    if nullable and value is None:
        return FieldResult(FieldState.PRESENT_VALID, None)
    if isinstance(value, bool) or not isinstance(value, int):
        return _invalid(name, "an integer", value)
    if minimum is not None and value < minimum:
        return FieldResult(FieldState.PRESENT_INVALID, error=f"{name} is below minimum {minimum}")
    if maximum is not None and value > maximum:
        return FieldResult(FieldState.PRESENT_INVALID, error=f"{name} exceeds maximum {maximum}")
    return FieldResult(FieldState.PRESENT_VALID, value)


def read_optional_sequence(
    raw: Mapping[str, Any], name: str, *, nullable: bool = False
) -> FieldResult[list[Any] | None]:
    if name not in raw:
        return _absent()
    value = raw[name]
    if nullable and value is None:
        return FieldResult(FieldState.PRESENT_VALID, None)
    if not isinstance(value, list):
        return _invalid(name, "a list", value)
    return FieldResult(FieldState.PRESENT_VALID, value)


def read_optional_mapping(
    raw: Mapping[str, Any], name: str, *, nullable: bool = False
) -> FieldResult[Mapping[str, Any] | None]:
    if name not in raw:
        return _absent()
    value = raw[name]
    if nullable and value is None:
        return FieldResult(FieldState.PRESENT_VALID, None)
    if not isinstance(value, Mapping):
        return _invalid(name, "an object", value)
    return FieldResult(FieldState.PRESENT_VALID, value)


def collect_invalid(*fields: FieldResult[Any]) -> list[str]:
    return [field.error or "invalid field" for field in fields if field.is_invalid]


def require_valid(field: FieldResult[T], name: str, errors: list[str]) -> T | None:
    if field.is_invalid:
        errors.append(field.error or f"{name} is invalid")
        return None
    return field.value if field.is_valid else None
