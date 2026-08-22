"""Small immutable projection primitives used at authority boundaries.

The runtime keeps complete canonical state internally, but values handed to a
provider must not be writable handles into that state.  ``FrozenMapping`` is a
deliberately small immutable Mapping implementation whose values are
recursively frozen.  It is also deepcopy-safe, which keeps the checkpoint host
compatible with the immutable model.
"""

from __future__ import annotations

from collections.abc import Mapping, Iterator
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from typing import Any, TypeVar


K = TypeVar("K")
V = TypeVar("V")


class FrozenMapping(Mapping[K, V]):
    """A recursively immutable, deepcopy-safe mapping.

    The backing representation is a tuple rather than a dict.  This avoids
    exposing a mutable private dictionary through an otherwise read-only
    interface and is sufficient for the small structured projections in Phase
    B.
    """

    __slots__ = ("_items",)

    def __init__(self, value: Mapping[K, V] | None = None, **kwargs: V) -> None:
        items = list((value or {}).items())
        items.extend(kwargs.items())
        object.__setattr__(
            self,
            "_items",
            tuple((key, freeze_value(item)) for key, item in items),
        )

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

    def __deepcopy__(self, memo: dict[int, Any]) -> "FrozenMapping[K, V]":
        memo[id(self)] = self
        return self

    def __copy__(self) -> "FrozenMapping[K, V]":
        return self


def freeze_value(value: Any) -> Any:
    """Recursively convert mutable containers to immutable projections."""

    if isinstance(value, FrozenMapping):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        # The Phase B structured dataclasses crossing this boundary are
        # frozen and contain only immutable fields.  Preserve their typed
        # identity; converting them to mappings would destroy the public
        # CompletionRequirement/GoalOutcome contract.  Mutable dataclasses
        # are defensively projected as mappings.
        params = getattr(type(value), "__dataclass_params__", None)
        if params is not None and getattr(params, "frozen", False):
            return value
        return freeze_value(asdict(value))
    if isinstance(value, Mapping):
        return FrozenMapping(value)
    if isinstance(value, list):
        return tuple(freeze_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(freeze_value(item) for item in value)
    return value


def isolated_copy(value: Any) -> Any:
    """Make a detached mutable snapshot for read-only audit projections."""

    return deepcopy(value)


def thaw_value(value: Any) -> Any:
    """Return a detached, ordinary-container view for human-readable traces."""

    if isinstance(value, FrozenMapping):
        return {key: thaw_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [thaw_value(item) for item in value]
    if isinstance(value, list):
        return [thaw_value(item) for item in value]
    if isinstance(value, Mapping):
        return {key: thaw_value(item) for key, item in value.items()}
    return value
