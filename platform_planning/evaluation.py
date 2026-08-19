from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .service import TaskPlanningService


def _get_path(payload: Any, path: str) -> Any:
    current = payload
    for part in path.split("."):
        if isinstance(current, list):
            current = current[int(part)]
        elif isinstance(current, dict):
            current = current[part]
        else:
            current = getattr(current, part)
    return current


def evaluate_task_planning(service: TaskPlanningService, cases_path: str | Path) -> dict[str, Any]:
    cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    rows = []
    passed = 0
    for case in cases:
        result = service.plan(case["query"])
        payload = result.model_dump(mode="json")
        checks = []
        checks.append({"path": "valid", "expected": case["expected_valid"], "actual": result.valid})
        for path, expected in (case.get("expected") or {}).items():
            try:
                actual = _get_path(payload, path)
                ok = actual == expected
                error = None
            except Exception as exc:
                actual = None
                ok = False
                error = str(exc)
            checks.append({"path": path, "expected": expected, "actual": actual, "ok": ok, "error": error})
        unresolved_expected = set(case.get("expected_unresolved") or [])
        if unresolved_expected:
            actual_unresolved = set(result.unresolved_fields)
            checks.append(
                {
                    "path": "unresolved_fields",
                    "expected": sorted(unresolved_expected),
                    "actual": sorted(actual_unresolved),
                    "ok": unresolved_expected.issubset(actual_unresolved),
                    "error": None,
                }
            )
        for check in checks:
            check.setdefault("ok", check["actual"] == check["expected"])
            check.setdefault("error", None)
        ok = all(item["ok"] for item in checks)
        passed += int(ok)
        rows.append({"id": case["id"], "ok": ok, "checks": checks})
    total = len(rows)
    return {
        "case_count": total,
        "passed": passed,
        "case_accuracy": (passed / total) if total else 0.0,
        "cases": rows,
    }
