"""Explicit safe JSON codec for durable Runtime checkpoints.

The decoder never imports a type named by checkpoint data.  Exact trusted
classes are registered from a fixed V2-local module allow-list at import time.
Unknown tagged types fail closed.
"""
from __future__ import annotations

import importlib
import json
import math
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from ..agent.immutable import FrozenMapping

_ALLOWED_MODULES = (
    "deploy_ci_cloud_agentv2.agent.budgets",
    "deploy_ci_cloud_agentv2.agent.contracts",
    "deploy_ci_cloud_agentv2.agent.decisions",
    "deploy_ci_cloud_agentv2.agent.evidence",
    "deploy_ci_cloud_agentv2.agent.goals",
    "deploy_ci_cloud_agentv2.agent.identity",
    "deploy_ci_cloud_agentv2.agent.outcomes",
    "deploy_ci_cloud_agentv2.agent.principles",
    "deploy_ci_cloud_agentv2.agent.provenance",
    "deploy_ci_cloud_agentv2.agent.results",
    "deploy_ci_cloud_agentv2.agent.state",
    "deploy_ci_cloud_agentv2.safety.approval",
    "deploy_ci_cloud_agentv2.safety.locks",
    "deploy_ci_cloud_agentv2.safety.write_transaction",
    "deploy_ci_cloud_agentv2.verification.results",
)

_TYPES: dict[str, type] = {}
_ENUMS: dict[str, type[Enum]] = {}
for module_name in _ALLOWED_MODULES:
    module = importlib.import_module(module_name)
    for name, value in vars(module).items():
        if not isinstance(value, type) or value.__module__ != module_name:
            continue
        key = f"{module_name}:{name}"
        if issubclass(value, Enum):
            _ENUMS[key] = value
        elif is_dataclass(value):
            _TYPES[key] = value


class CheckpointCodecError(ValueError):
    pass


class LangGraphCheckpointSerializer:
    """LangGraph serializer backed by the V2 allow-listed checkpoint codec.

    LangGraph's default msgpack serializer cannot encode V2's typed Runtime
    projection (for example ``CurrentRequestContext``).  This adapter keeps
    the graph checkpointer on the same explicit, non-pickle codec used by the
    durable checkpoint boundary.  It is deliberately small: LangGraph only
    requires ``dumps_typed`` and ``loads_typed``.
    """

    def dumps_typed(self, value: Any) -> tuple[str, bytes]:
        try:
            encoded = encode(value)
            payload = json.dumps(
                encoded,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        except Exception as exc:
            if isinstance(exc, CheckpointCodecError):
                raise
            raise CheckpointCodecError("could not encode LangGraph checkpoint value") from exc
        return "json", payload

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        type_name, payload = data
        if type_name != "json":
            raise CheckpointCodecError(
                f"unsupported LangGraph checkpoint encoding: {type_name}"
            )
        try:
            return decode(json.loads(payload.decode("utf-8")))
        except Exception as exc:
            if isinstance(exc, CheckpointCodecError):
                raise
            raise CheckpointCodecError("could not decode LangGraph checkpoint value") from exc


def encode(value: Any) -> Any:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise CheckpointCodecError("non-finite float in checkpoint")
        return value
    if isinstance(value, datetime):
        return {"$datetime": value.isoformat()}
    if isinstance(value, Enum):
        key = f"{type(value).__module__}:{type(value).__name__}"
        if key not in _ENUMS:
            raise CheckpointCodecError(f"unregistered enum: {key}")
        return {"$enum": key, "value": encode(value.value)}
    # LangGraph stores the value raised by ``interrupt()`` as a small typed
    # marker in checkpoint writes.  It is the only third-party runtime value
    # admitted here, and it is reduced to its two scalar/data fields rather
    # than retaining the object or importing a type from checkpoint data.
    if (
        type(value).__module__ == "langgraph.types"
        and type(value).__name__ == "Interrupt"
    ):
        return {
            "$langgraph_interrupt": {
                "value": encode(value.value),
                "id": encode(value.id),
            }
        }
    if isinstance(value, FrozenMapping):
        return {"$frozen_map": [[key, encode(item)] for key, item in value.items()]}
    if type(value) is dict:
        rows = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise CheckpointCodecError("checkpoint mapping keys must be strings")
            rows.append([key, encode(item)])
        return {"$map": rows}
    if type(value) is tuple:
        return {"$tuple": [encode(item) for item in value]}
    if type(value) is list:
        return {"$list": [encode(item) for item in value]}
    if type(value) is frozenset:
        encoded = [encode(item) for item in value]
        encoded.sort(key=repr)
        return {"$frozenset": encoded}
    if is_dataclass(value) and not isinstance(value, type):
        key = f"{type(value).__module__}:{type(value).__name__}"
        cls = _TYPES.get(key)
        if cls is not type(value):
            raise CheckpointCodecError(f"unregistered dataclass: {key}")
        return {
            "$type": key,
            "fields": {
                item.name: encode(getattr(value, item.name))
                for item in fields(value)
                if item.init
            },
        }
    if isinstance(value, Mapping):
        # Do not silently trust arbitrary mapping implementations.
        raise CheckpointCodecError(f"unsupported mapping type: {type(value).__name__}")
    raise CheckpointCodecError(f"unsupported checkpoint value: {type(value).__name__}")


def decode(value: Any) -> Any:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise CheckpointCodecError("non-finite float in checkpoint")
        return value
    if type(value) is list:
        raise CheckpointCodecError("untagged list in checkpoint")
    if type(value) is not dict:
        raise CheckpointCodecError(f"unsupported encoded checkpoint value: {type(value).__name__}")
    if set(value) == {"$datetime"}:
        try:
            return datetime.fromisoformat(value["$datetime"])
        except Exception as exc:
            raise CheckpointCodecError("invalid datetime in checkpoint") from exc
    if set(value) == {"$enum", "value"}:
        cls = _ENUMS.get(value["$enum"])
        if cls is None:
            raise CheckpointCodecError("unregistered enum tag")
        try:
            return cls(decode(value["value"]))
        except Exception as exc:
            raise CheckpointCodecError("invalid enum value") from exc
    if set(value) == {"$langgraph_interrupt"}:
        raw = value["$langgraph_interrupt"]
        if type(raw) is not dict or set(raw) != {"value", "id"}:
            raise CheckpointCodecError("invalid LangGraph interrupt encoding")
        try:
            from langgraph.types import Interrupt

            interrupt_type = Interrupt
            interrupt_value = decode(raw["value"])
            interrupt_id = decode(raw["id"])
            if type(interrupt_id) is not str:
                raise CheckpointCodecError("invalid LangGraph interrupt id")
            return interrupt_type(interrupt_value, id=interrupt_id)
        except CheckpointCodecError:
            raise
        except Exception as exc:
            raise CheckpointCodecError("invalid LangGraph interrupt payload") from exc
    if set(value) == {"$tuple"}:
        if type(value["$tuple"]) is not list:
            raise CheckpointCodecError("invalid tuple encoding")
        return tuple(decode(item) for item in value["$tuple"])
    if set(value) == {"$list"}:
        if type(value["$list"]) is not list:
            raise CheckpointCodecError("invalid list encoding")
        return [decode(item) for item in value["$list"]]
    if set(value) == {"$frozenset"}:
        if type(value["$frozenset"]) is not list:
            raise CheckpointCodecError("invalid frozenset encoding")
        return frozenset(decode(item) for item in value["$frozenset"])
    if set(value) == {"$map"}:
        return _decode_map_rows(value["$map"], frozen=False)
    if set(value) == {"$frozen_map"}:
        return _decode_map_rows(value["$frozen_map"], frozen=True)
    if set(value) == {"$type", "fields"}:
        cls = _TYPES.get(value["$type"])
        if cls is None:
            raise CheckpointCodecError("unregistered dataclass tag")
        raw_fields = value["fields"]
        if type(raw_fields) is not dict:
            raise CheckpointCodecError("invalid dataclass fields")
        allowed = {item.name for item in fields(cls) if item.init}
        if set(raw_fields) != allowed:
            raise CheckpointCodecError(
                f"dataclass field set mismatch for {value['$type']}"
            )
        decoded = {name: decode(item) for name, item in raw_fields.items()}
        try:
            return cls(**decoded)
        except Exception as exc:
            raise CheckpointCodecError(f"invalid dataclass payload for {value['$type']}") from exc
    raise CheckpointCodecError("unknown checkpoint tag")


def _decode_map_rows(rows: Any, *, frozen: bool) -> Any:
    if type(rows) is not list:
        raise CheckpointCodecError("invalid mapping encoding")
    output: dict[str, Any] = {}
    for row in rows:
        if type(row) is not list or len(row) != 2 or not isinstance(row[0], str):
            raise CheckpointCodecError("invalid mapping row")
        key = row[0]
        if key in output:
            raise CheckpointCodecError("duplicate mapping key")
        output[key] = decode(row[1])
    return FrozenMapping(output) if frozen else output
