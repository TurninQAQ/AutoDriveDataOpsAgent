"""Deterministic identity and scope provenance for normalized READ results."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from .immutable import thaw_value


class ScopeKind(str, Enum):
    PLATFORM = "PLATFORM"
    TASK = "TASK"
    QUERY = "QUERY"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ObservationScope:
    kind: ScopeKind
    identity: str | None = None

    def __post_init__(self) -> None:
        if self.kind in {ScopeKind.PLATFORM, ScopeKind.UNKNOWN} and self.identity is not None:
            raise ValueError(f"{self.kind.value} scope cannot carry an identity")
        if self.kind in {ScopeKind.TASK, ScopeKind.QUERY}:
            if not isinstance(self.identity, str) or not self.identity.strip():
                raise ValueError(f"{self.kind.value} scope requires a non-empty identity")
            object.__setattr__(self, "identity", self.identity.strip())


class IdentityStatus(str, Enum):
    MATCHED = "MATCHED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    MISSING = "MISSING"
    CONFLICT = "CONFLICT"


class ScopeStatus(str, Enum):
    MATCHED = "MATCHED"
    MISSING = "MISSING"
    CONFLICT = "CONFLICT"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ObservationProvenance:
    source_tool: str
    arguments_fingerprint: str
    requested_scope: ObservationScope
    observed_scope: ObservationScope
    requested_identity: str | None
    observed_identity: str | None
    identity_status: IdentityStatus
    scope_status: ScopeStatus

    @property
    def requested_target(self) -> str:
        return self.requested_identity or "platform"

    @property
    def observed_target(self) -> str | None:
        return self.observed_identity


def normalized_arguments_fingerprint(arguments: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        thaw_value(dict(arguments)),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def canonical_tool_call_fingerprint(
    source_tool: str, arguments: Mapping[str, Any]
) -> str:
    """Fingerprint the one canonical representation used by audit/evidence.

    Callers pass normalized arguments.  The small normalization fallback keeps
    direct provenance helpers deterministic for valid calls, while Runtime
    validation remains the authority for accepting a call.
    """

    normalized = _normalize_for_provenance(source_tool, arguments)
    encoded = json.dumps(
        {"tool_name": source_tool, "arguments": thaw_value(normalized)},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def requested_scope_for(source_tool: str, arguments: Mapping[str, Any]) -> ObservationScope:
    if source_tool in {"get_task_detail", "diagnose_task"}:
        return ObservationScope(ScopeKind.TASK, str(arguments["task_name"]).strip())
    if source_tool == "get_queue_state":
        task_name = arguments.get("task_name")
        return (
            ObservationScope(ScopeKind.PLATFORM)
            if task_name is None
            else ObservationScope(ScopeKind.TASK, str(task_name).strip())
        )
    if source_tool == "search_knowledge":
        return ObservationScope(ScopeKind.QUERY, str(arguments["query"]).strip())
    return ObservationScope(ScopeKind.PLATFORM)


def build_provenance(
    source_tool: str,
    arguments: Mapping[str, Any],
    result: Any | None,
) -> ObservationProvenance:
    arguments = _normalize_for_provenance(source_tool, arguments)
    requested_scope = requested_scope_for(source_tool, arguments)
    requested_identity = requested_scope.identity
    observed_scope = getattr(result, "observed_scope", ObservationScope(ScopeKind.UNKNOWN))
    observed_identity = observed_scope.identity

    if requested_scope.kind is ScopeKind.PLATFORM and observed_scope.kind is ScopeKind.PLATFORM:
        scope_status = ScopeStatus.MATCHED
    elif requested_scope.kind is ScopeKind.TASK and observed_scope.kind is ScopeKind.TASK:
        scope_status = (
            ScopeStatus.MATCHED
            if requested_identity == observed_identity
            else ScopeStatus.CONFLICT
        )
    elif requested_scope.kind is ScopeKind.QUERY and observed_scope.kind is ScopeKind.QUERY:
        scope_status = (
            ScopeStatus.MATCHED
            if requested_identity == observed_identity
            else ScopeStatus.CONFLICT
        )
    elif observed_scope.kind is ScopeKind.UNKNOWN:
        scope_status = ScopeStatus.MISSING
    else:
        scope_status = ScopeStatus.CONFLICT

    if source_tool == "get_gpu_pool":
        identity_status = IdentityStatus.NOT_APPLICABLE
    elif requested_identity is None and observed_identity is None:
        identity_status = IdentityStatus.NOT_APPLICABLE
    elif observed_identity is None:
        identity_status = IdentityStatus.MISSING
    elif requested_identity == observed_identity:
        identity_status = IdentityStatus.MATCHED
    else:
        identity_status = IdentityStatus.CONFLICT

    return ObservationProvenance(
        source_tool=source_tool,
        arguments_fingerprint=canonical_tool_call_fingerprint(source_tool, arguments),
        requested_scope=requested_scope,
        observed_scope=observed_scope,
        requested_identity=requested_identity,
        observed_identity=observed_identity,
        identity_status=identity_status,
        scope_status=scope_status,
    )


def _normalize_for_provenance(source_tool: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Keep direct provenance callers consistent with the Runtime boundary."""

    normalized = dict(arguments)
    for key in ("task_name", "query"):
        value = normalized.get(key)
        if isinstance(value, str):
            normalized[key] = value.strip()
    if source_tool == "search_knowledge" and "top_k" not in normalized:
        normalized["top_k"] = 5
    if source_tool == "get_queue_state" and "task_name" not in normalized:
        normalized["task_name"] = None
    return normalized
