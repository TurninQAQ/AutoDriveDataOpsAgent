"""Closed immutable values used at Runtime authority boundaries.

The Runtime never treats an arbitrary Python object as canonical data. Values
crossing an ingress boundary are detached and reduced to the small
``CanonicalValue`` domain below. This is intentionally stricter than
``deepcopy``: copying an object does not make an SDK object or a byte buffer a
deterministic audit value.
"""

from __future__ import annotations

from collections.abc import Mapping, Iterator
from copy import deepcopy
from dataclasses import fields, is_dataclass
from datetime import date, datetime, time
from enum import Enum
import math
from typing import Any, TypeAlias, TypeVar
from uuid import UUID


K = TypeVar("K")
V = TypeVar("V")


class CanonicalizationError(ValueError):
    """A value is outside the closed Runtime canonical value domain."""


class FrozenMapping(Mapping[K, V]):
    """A recursively immutable mapping with string keys."""

    __slots__ = ("_items",)

    def __init__(self, value: Mapping[K, V] | None = None, **kwargs: V) -> None:
        if value is not None and not isinstance(value, Mapping):
            raise CanonicalizationError("FrozenMapping requires a mapping")
        items: list[tuple[Any, Any]] = list(value.items()) if value is not None else []
        items.extend(kwargs.items())
        frozen: list[tuple[str, Any]] = []
        for key, item in items:
            if not isinstance(key, str):
                raise CanonicalizationError("canonical mapping keys must be strings")
            frozen.append((key, _freeze_internal(item)))
        object.__setattr__(self, "_items", tuple(frozen))

    def __setattr__(self, name: str, value: object) -> None:
        raise TypeError("FrozenMapping is immutable")

    def __getitem__(self, key: K) -> V:
        for item_key, item_value in self._items:
            if item_key == key:
                return item_value  # type: ignore[return-value]
        raise KeyError(key)

    def __iter__(self) -> Iterator[K]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"FrozenMapping({dict(self._items)!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            return dict(self.items()) == dict(other.items())
        return NotImplemented

    def __deepcopy__(self, memo: dict[int, Any]) -> "FrozenMapping[K, V]":
        memo[id(self)] = self
        return self

    def __copy__(self) -> "FrozenMapping[K, V]":
        return self


CanonicalValue: TypeAlias = (
    type(None)
    | bool
    | int
    | float
    | str
    | tuple["CanonicalValue", ...]
    | FrozenMapping[str, "CanonicalValue"]
)


def freeze_value(value: Any) -> Any:
    """Canonicalize a Runtime-owned value and reject unknown leaves."""

    if isinstance(value, FrozenMapping):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        params = getattr(type(value), "__dataclass_params__", None)
        if params is not None and getattr(params, "frozen", False):
            clone = deepcopy(value)
            try:
                for item in fields(value):
                    object.__setattr__(clone, item.name, _freeze_internal(getattr(clone, item.name)))
                return clone
            except (AttributeError, TypeError, CanonicalizationError):
                return FrozenMapping(
                    {item.name: _freeze_internal(getattr(value, item.name)) for item in fields(value)}
                )
        return FrozenMapping(
            {item.name: _freeze_internal(getattr(value, item.name)) for item in fields(value)}
        )
    return _freeze_internal(value)


def isolated_copy(value: Any) -> Any:
    """Make a detached mutable projection from canonical data."""

    return deepcopy(value)


def canonical_snapshot(value: Any) -> Any:
    """Detach external JSON-like data and return a closed immutable snapshot.

    External tool payloads are intentionally strict: sets, bytes, buffers and
    arbitrary objects are not JSON-like and must be normalized by the adapter
    before ingress. Dataclasses and supported scalar special types are
    deterministically projected to canonical data.
    """

    if isinstance(value, FrozenMapping):
        # FrozenMapping construction already proved every reachable child is
        # canonical; retaining this immutable object avoids a second ingress
        # representation for ToolObservation.data.
        return value
    try:
        detached = deepcopy(value)
    except Exception as exc:  # pragma: no cover - defensive boundary
        raise CanonicalizationError(f"value cannot be detached: {type(value).__name__}") from exc
    return _canonicalize(detached, allow_sets=False, dataclass_as_mapping=True)


def _freeze_internal(value: Any) -> Any:
    return _canonicalize(value, allow_sets=False, dataclass_as_mapping=False)


def _canonicalize(value: Any, *, allow_sets: bool, dataclass_as_mapping: bool) -> Any:
    # bool must precede int because bool is an int subclass.
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalizationError("non-finite floats are not canonical values")
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, Enum):
        return _canonicalize(value.value, allow_sets=allow_sets, dataclass_as_mapping=dataclass_as_mapping)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise CanonicalizationError(f"unsupported binary value: {type(value).__name__}")
    if callable(value):
        raise CanonicalizationError("callable values are not canonical")
    if is_dataclass(value) and not isinstance(value, type):
        params = getattr(type(value), "__dataclass_params__", None)
        if not dataclass_as_mapping and params is not None and getattr(params, "frozen", False):
            clone = deepcopy(value)
            try:
                for item in fields(value):
                    object.__setattr__(
                        clone,
                        item.name,
                        _canonicalize(
                            getattr(clone, item.name),
                            allow_sets=allow_sets,
                            dataclass_as_mapping=False,
                        ),
                    )
                return clone
            except (AttributeError, TypeError, CanonicalizationError):
                # A typed frozen dataclass with an unsupported child is not a
                # canonical value merely because its outer shell is frozen.
                raise CanonicalizationError(
                    f"frozen dataclass {type(value).__name__} contains a non-canonical field"
                )
        return FrozenMapping(
            {
                item.name: _canonicalize(
                    getattr(value, item.name),
                    allow_sets=allow_sets,
                    dataclass_as_mapping=True,
                )
                for item in fields(value)
            }
        )
    if isinstance(value, Mapping):
        items: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalizationError("canonical mapping keys must be strings")
            items[key] = _canonicalize(
                item, allow_sets=allow_sets, dataclass_as_mapping=dataclass_as_mapping
            )
        return FrozenMapping(items)
    if isinstance(value, list):
        return tuple(
            _canonicalize(item, allow_sets=allow_sets, dataclass_as_mapping=dataclass_as_mapping)
            for item in value
        )
    if isinstance(value, tuple):
        return tuple(
            _canonicalize(item, allow_sets=allow_sets, dataclass_as_mapping=dataclass_as_mapping)
            for item in value
        )
    if isinstance(value, (set, frozenset)):
        if not allow_sets:
            raise CanonicalizationError("set values must be explicitly normalized before ingress")
        raise CanonicalizationError("sets are not part of the canonical value domain")
    raise CanonicalizationError(f"unsupported canonical value: {type(value).__name__}")


def thaw_value(value: Any) -> Any:
    """Return a detached ordinary-container projection for read APIs."""

    if isinstance(value, FrozenMapping):
        return {key: thaw_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_value(item) for item in value]
    if isinstance(value, list):
        return [thaw_value(item) for item in value]
    if isinstance(value, Mapping):
        return {key: thaw_value(item) for key, item in value.items()}
    return value
