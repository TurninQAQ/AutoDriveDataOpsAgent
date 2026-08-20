"""Deterministic argument-contract matching shared by evaluation adapters.

Legacy evaluation cases only supplied ``expected_arguments``.  Those values
remain subset requirements.  V1.4.4 cases may additionally supply an
``argument_contract`` with an explicit matcher per field.  Keeping both
semantics here prevents the aligned evaluator and DeepEval adapter from
quietly scoring the same case differently.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from platform_mcp.server import WRITE_TOOL_NAMES


VALID_MATCHERS = frozenset({"exact", "subset", "present", "non_empty", "one_of", "range"})
VALID_CASE_CATEGORIES = frozenset({
    "catalog",
    "diagnosis",
    "health",
    "knowledge",
    "task",
    "write",
    "no_tool",
    "platform_read",
    "rag_required",
    "rag_optional_with_platform",
    "static_knowledge",
    "live_gpu_state",
    "live_task_state",
    "gpu_diagnosis",
    "named_task_diagnosis",
    "hybrid_live_knowledge",
    "task_planning",
})


class ArgumentContractError(ValueError):
    """Raised when an evaluation case has an invalid argument contract."""


def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArgumentContractError(f"{label} must be an object")
    return value


def _subset_match(actual: Any, expected: Any) -> bool:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return False
        return all(key in actual and _subset_match(actual[key], value) for key, value in expected.items())
    return actual == expected


def legacy_subset_match(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    """Preserve the V1.1 ``expected_arguments`` recursive subset semantics."""

    return _subset_match(actual, expected)


def normalize_legacy_expected_arguments(expected_arguments: Any) -> dict[str, dict[str, Any]]:
    if expected_arguments is None:
        return {}
    expected = _as_mapping(expected_arguments, "expected_arguments")
    normalized: dict[str, dict[str, Any]] = {}
    for tool_name, value in expected.items():
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ArgumentContractError("expected_arguments tool names must be non-empty strings")
        if not isinstance(value, Mapping):
            raise ArgumentContractError(f"expected_arguments[{tool_name}] must be an object")
        normalized[tool_name] = dict(value)
    return normalized


def validate_argument_contract(argument_contract: Any) -> None:
    if argument_contract is None:
        return
    contract = _as_mapping(argument_contract, "argument_contract")
    for tool_name, fields in contract.items():
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ArgumentContractError("argument_contract tool names must be non-empty strings")
        fields = _as_mapping(fields, f"argument_contract[{tool_name}]")
        for field_name, rule in fields.items():
            if not isinstance(field_name, str) or not field_name.strip():
                raise ArgumentContractError("argument contract field names must be non-empty strings")
            rule = _as_mapping(rule, f"argument_contract[{tool_name}][{field_name}]")
            matcher = rule.get("match")
            if matcher not in VALID_MATCHERS:
                raise ArgumentContractError(
                    f"unsupported argument matcher {matcher!r}; expected one of {sorted(VALID_MATCHERS)}"
                )
            if matcher in {"exact", "subset"} and "value" not in rule:
                raise ArgumentContractError(f"{matcher} matcher requires value")
            if matcher == "one_of":
                values = rule.get("values", rule.get("value"))
                if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
                    raise ArgumentContractError("one_of matcher requires a non-empty values list")
            if matcher == "range":
                if "min" not in rule and "max" not in rule:
                    bounds = rule.get("value")
                    if not isinstance(bounds, Mapping) or ("min" not in bounds and "max" not in bounds):
                        raise ArgumentContractError("range matcher requires min/max bounds")


def validate_tool_case(case: Mapping[str, Any], *, index: int | None = None) -> None:
    prefix = f"case {index}: " if index is not None else ""
    query = case.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ArgumentContractError(prefix + "query must be a non-empty string")
    case_id = case.get("id")
    if case_id is not None and (not isinstance(case_id, str) or not case_id.strip()):
        raise ArgumentContractError(prefix + "id must be a non-empty string")
    category = case.get("category")
    if category is not None and category not in VALID_CASE_CATEGORIES:
        raise ArgumentContractError(prefix + f"unsupported category: {category!r}")
    for field in ("required_tools", "optional_tools", "forbidden_tools", "required_order"):
        value = case.get(field, [])
        if value is None:
            continue
        if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
            raise ArgumentContractError(prefix + f"{field} must be a list of tool-name strings")
    required = set(case.get("required_tools") or [])
    optional = set(case.get("optional_tools") or [])
    forbidden = set(case.get("forbidden_tools") or [])
    if required & optional:
        raise ArgumentContractError(prefix + "required_tools and optional_tools overlap")
    if required & forbidden:
        raise ArgumentContractError(prefix + "required_tools and forbidden_tools overlap")
    if optional & forbidden:
        raise ArgumentContractError(prefix + "optional_tools and forbidden_tools overlap")
    order = list(case.get("required_order") or [])
    if not set(order) <= required | optional:
        raise ArgumentContractError(prefix + "required_order contains a tool outside required/optional_tools")
    if not isinstance(case.get("expected_intent", ""), str):
        raise ArgumentContractError(prefix + "expected_intent must be a string")
    normalize_legacy_expected_arguments(case.get("expected_arguments"))
    validate_argument_contract(case.get("argument_contract"))
    if case.get("category") == "write":
        # A write case may describe read evidence, but direct mutation tools are
        # never an allowed planner output regardless of what its Golden lists.
        if required & set(WRITE_TOOL_NAMES) or optional & set(WRITE_TOOL_NAMES):
            raise ArgumentContractError(prefix + "write cases cannot require or allow direct write tools")


def validate_tool_cases(cases: Sequence[Mapping[str, Any]]) -> None:
    seen: set[str] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise ArgumentContractError(f"case {index}: expected an object")
        case_id = case.get("id")
        if case_id is not None:
            if case_id in seen:
                raise ArgumentContractError(f"duplicate case id: {case_id}")
            seen.add(str(case_id))
        validate_tool_case(case, index=index)


def _value_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (Mapping, Sequence)) and not isinstance(value, (str, bytes)):
        return bool(value)
    return True


def _rule_value(rule: Mapping[str, Any]) -> Any:
    if "value" in rule:
        return rule["value"]
    return None


def _match_rule(actual: Any, rule: Mapping[str, Any]) -> tuple[bool, bool, str | None]:
    matcher = rule["match"]
    present = _value_present(actual)
    if matcher == "exact":
        return actual == _rule_value(rule), present, "wrong_exact" if present and actual != _rule_value(rule) else None
    if matcher == "subset":
        return _subset_match(actual, _rule_value(rule)), present, None
    if matcher == "present":
        # The contract is structural: a present field may legitimately carry
        # null/false/zero.  ``non_empty`` is the stricter matcher.
        return True, True, None
    if matcher == "non_empty":
        return present, present, None
    if matcher == "one_of":
        values = rule.get("values", _rule_value(rule))
        return actual in values, present, None
    bounds = rule.get("value") if isinstance(rule.get("value"), Mapping) else rule
    minimum = bounds.get("min")
    maximum = bounds.get("max")
    try:
        ok = (minimum is None or actual >= minimum) and (maximum is None or actual <= maximum)
    except TypeError:
        ok = False
    return ok, present, None


def _actual_call_parts(call: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(call, Mapping):
        return str(call.get("name") or ""), dict(call.get("arguments") or {})
    return str(getattr(call, "name", "") or ""), dict(getattr(call, "arguments", {}) or {})


def evaluate_argument_contract(
    actual_calls: Sequence[Any],
    *,
    expected_arguments: Mapping[str, Any] | None = None,
    argument_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one case and return deterministic plus diagnostic metrics.

    A V1.4.4 contract overrides the legacy rule for the same tool.  Legacy
    non-empty objects are one recursive subset requirement, which preserves
    the historical denominator.  New field contracts count each declared
    field, allowing presence, exactness and semantic-search structure to be
    reported separately.
    """

    legacy = normalize_legacy_expected_arguments(expected_arguments)
    contract = dict(argument_contract or {})
    validate_argument_contract(contract)
    calls_by_tool: dict[str, list[dict[str, Any]]] = {}
    for call in actual_calls:
        name, arguments = _actual_call_parts(call)
        if name:
            calls_by_tool.setdefault(name, []).append(arguments)

    details: list[dict[str, Any]] = []
    for tool_name in dict.fromkeys([*legacy.keys(), *contract.keys()]):
        matching = calls_by_tool.get(tool_name, [])
        if tool_name in contract:
            fields = contract[tool_name]
            if not fields:
                details.append({"tool": tool_name, "kind": "tool_presence", "ok": bool(matching), "present": bool(matching)})
            else:
                for field_name, rule in fields.items():
                    rule = dict(rule)
                    actual_values = [args[field_name] for args in matching if field_name in args]
                    if actual_values:
                        outcomes = [_match_rule(value, rule) for value in actual_values]
                        ok = any(item[0] for item in outcomes)
                        present = any(item[1] for item in outcomes)
                        reason = next((item[2] for item in outcomes if item[2]), None)
                    else:
                        ok = False
                        present = False
                        reason = "missing_argument"
                    details.append({
                        "tool": tool_name,
                        "field": field_name,
                        "matcher": rule["match"],
                        "expected": rule,
                        "actual": actual_values,
                        "ok": ok,
                        "present": present,
                        "failure_type": reason or (None if ok else "argument_mismatch"),
                    })
        elif legacy[tool_name]:
            matching = calls_by_tool.get(tool_name, [])
            ok = any(legacy_subset_match(args, legacy[tool_name]) for args in matching)
            details.append({
                "tool": tool_name,
                "kind": "legacy_subset",
                "expected": legacy[tool_name],
                "actual": matching,
                "ok": ok,
                "present": bool(matching),
                "failure_type": None if ok else ("missing_argument" if not matching else "argument_mismatch"),
            })
        else:
            details.append({
                "tool": tool_name,
                "kind": "tool_presence",
                "ok": bool(matching),
                "present": bool(matching),
            })

    total = len(details)
    hits = sum(int(item["ok"]) for item in details)
    presence_hits = sum(int(item.get("present", False)) for item in details)
    exact = [item for item in details if item.get("matcher") == "exact"]
    exact_hits = sum(int(item["ok"]) for item in exact)
    return {
        "ok": hits == total,
        "hits": hits,
        "total": total,
        "contract_accuracy": hits / total if total else 1.0,
        "presence_hits": presence_hits,
        "presence_total": total,
        "presence_coverage": presence_hits / total if total else 1.0,
        "exact_hits": exact_hits,
        "exact_total": len(exact),
        "exact_accuracy": exact_hits / len(exact) if exact else 1.0,
        "details": details,
        "missing_arguments": [item for item in details if item.get("failure_type") == "missing_argument"],
        "wrong_exact_arguments": [item for item in details if item.get("matcher") == "exact" and not item.get("ok")],
    }
