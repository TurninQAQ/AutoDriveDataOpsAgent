from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


ScenarioCategory = Literal["read", "planning", "safe_auto", "hitl", "deny", "adversarial"]
SCENARIO_CATEGORIES = {"read", "planning", "safe_auto", "hitl", "deny", "adversarial"}
POLICY_MODES = {"AUTO", "HITL", "DENY"}
GOAL_STATES = {"SATISFIED", "IN_PROGRESS", "FAILED", "INCONCLUSIVE"}


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_string(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string or null")
    return value


def _string_list(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key, [])
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be a list of strings")
    return list(value)


def _dict_value(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return dict(value)


def _nonnegative_int(payload: dict[str, Any], key: str, default: int = 0) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class Scenario:
    """Frozen deterministic ground truth for one benchmark scenario."""

    id: str
    category: ScenarioCategory
    prompt: str
    fixture: str
    expected_intent: str
    expected_target: str | None = None
    expected_policy: str | None = None
    expected_goal: str | None = None
    goal_eval: bool = False
    risk_class: str | None = None
    expected_datasets: list[str] = field(default_factory=list)
    fixture_payload: dict[str, Any] = field(default_factory=dict)
    max_mutations: int = 0
    required_tools: list[str] = field(default_factory=list)
    allowed_optional_tools: list[str] = field(default_factory=list)
    safety_constraints: list[str] = field(default_factory=list)

    @classmethod
    def model_validate(cls, payload: Any) -> "Scenario":
        if not isinstance(payload, dict):
            raise ValueError("scenario must be an object")
        category = _required_string(payload, "category")
        if category not in SCENARIO_CATEGORIES:
            raise ValueError(f"invalid category={category!r}")
        goal_eval = payload.get("goal_eval", False)
        if not isinstance(goal_eval, bool):
            raise ValueError("goal_eval must be boolean")
        return cls(
            id=_required_string(payload, "id"),
            category=category,  # type: ignore[arg-type]
            prompt=_required_string(payload, "prompt"),
            fixture=_required_string(payload, "fixture"),
            expected_intent=_required_string(payload, "expected_intent"),
            expected_target=_optional_string(payload, "expected_target"),
            expected_policy=_optional_string(payload, "expected_policy"),
            expected_goal=_optional_string(payload, "expected_goal"),
            goal_eval=goal_eval,
            risk_class=_optional_string(payload, "risk_class"),
            expected_datasets=_string_list(payload, "expected_datasets"),
            fixture_payload=_dict_value(payload, "fixture_payload"),
            max_mutations=_nonnegative_int(payload, "max_mutations"),
            required_tools=_string_list(payload, "required_tools"),
            allowed_optional_tools=_string_list(payload, "allowed_optional_tools"),
            safety_constraints=_string_list(payload, "safety_constraints"),
        )

    def validate_contract(self) -> None:
        if self.expected_policy is not None and self.expected_policy not in POLICY_MODES:
            raise ValueError(f"{self.id}: invalid expected_policy={self.expected_policy!r}")
        if self.expected_goal is not None and self.expected_goal.upper() not in GOAL_STATES:
            raise ValueError(f"{self.id}: invalid expected_goal={self.expected_goal!r}")
        if self.goal_eval and self.expected_goal is None:
            raise ValueError(f"{self.id}: goal_eval scenarios require expected_goal")
        if self.risk_class is not None and self.risk_class not in {"AUTO_ELIGIBLE", "HITL_REQUIRED", "DENY_REQUIRED", "NONE"}:
            raise ValueError(f"{self.id}: invalid risk_class={self.risk_class!r}")
        if len(set(self.required_tools) & set(self.allowed_optional_tools)):
            raise ValueError(f"{self.id}: required_tools and allowed_optional_tools overlap")
        if self.category == "safe_auto" and self.expected_policy != "AUTO":
            raise ValueError(f"{self.id}: safe_auto scenarios must expect AUTO")
        if self.category == "hitl" and self.expected_policy != "HITL":
            raise ValueError(f"{self.id}: hitl scenarios must expect HITL")
        if self.category == "deny" and self.expected_policy != "DENY":
            raise ValueError(f"{self.id}: deny scenarios must expect DENY")
        if self.expected_policy == "AUTO" and self.max_mutations != 1:
            raise ValueError(f"{self.id}: AUTO scenarios must cap mutations at one")

    @property
    def effective_risk_class(self) -> str:
        if self.risk_class:
            return self.risk_class
        return {"AUTO": "AUTO_ELIGIBLE", "HITL": "HITL_REQUIRED", "DENY": "DENY_REQUIRED"}.get(self.expected_policy or "", "NONE")

    def signature(self) -> str:
        normalized = {
            "prompt": " ".join(self.prompt.casefold().split()),
            "fixture": self.fixture,
            "fixture_payload": self.fixture_payload,
            "expected_intent": self.expected_intent.upper(),
            "expected_target": self.expected_target,
            "expected_risk": self.effective_risk_class,
            "expected_goal": self.expected_goal.upper() if self.expected_goal else None,
            "expected_datasets": sorted(self.expected_datasets),
        }
        return hashlib.sha256(json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SafetyScenario:
    id: str
    family: str
    fixture: str
    expected_policy: str | None = None
    expected_goal: str | None = None
    expected_mutations: int = 0
    safety_invariants: list[str] = field(default_factory=list)

    @classmethod
    def model_validate(cls, payload: Any) -> "SafetyScenario":
        if not isinstance(payload, dict):
            raise ValueError("safety scenario must be an object")
        return cls(
            id=_required_string(payload, "id"),
            family=_required_string(payload, "family"),
            fixture=_required_string(payload, "fixture"),
            expected_policy=_optional_string(payload, "expected_policy"),
            expected_goal=_optional_string(payload, "expected_goal"),
            expected_mutations=_nonnegative_int(payload, "expected_mutations"),
            safety_invariants=_string_list(payload, "safety_invariants"),
        )

    def validate_contract(self) -> None:
        if self.expected_policy is not None and self.expected_policy not in POLICY_MODES:
            raise ValueError(f"{self.id}: invalid expected_policy={self.expected_policy!r}")
        if self.expected_goal is not None and self.expected_goal.upper() not in GOAL_STATES:
            raise ValueError(f"{self.id}: invalid expected_goal={self.expected_goal!r}")


def _load_jsonl(path: str | Path, model: Any) -> list[Any]:
    rows: list[Any] = []
    seen: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            try:
                item = model.model_validate(json.loads(raw))
                item.validate_contract()
            except Exception as exc:
                raise ValueError(f"Invalid {path}:{line_no}: {exc}") from exc
            if item.id in seen:
                raise ValueError(f"Duplicate scenario id: {item.id}")
            seen.add(item.id)
            rows.append(item)
    return rows


def load_scenarios(path: str | Path) -> list[Scenario]:
    return _load_jsonl(path, Scenario)


def load_safety_scenarios(path: str | Path) -> list[SafetyScenario]:
    return _load_jsonl(path, SafetyScenario)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_split(path: str | Path, *, expected_count: int | None = None) -> list[Scenario]:
    cases = load_scenarios(path)
    if expected_count is not None and len(cases) != expected_count:
        raise ValueError(f"{path}: expected {expected_count} scenarios, found {len(cases)}")
    return cases


def signature_overlap(left: list[Scenario], right: list[Scenario]) -> set[str]:
    return {case.signature() for case in left} & {case.signature() for case in right}
