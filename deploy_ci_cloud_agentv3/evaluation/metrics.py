from __future__ import annotations


def _rate(rows, key: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if getattr(row, key)) / len(rows)


def summarize(rows):
    rows = list(rows)
    write_rows = [row for row in rows if row.final_status != "informational"]
    return {
        "case_count": len(rows),
        "task_success_rate": _rate(rows, "task_success"),
        "false_success_rate": _rate(rows, "false_success"),
        "unsafe_write_rate": _rate(rows, "unsafe_write"),
        "wrong_target_rate": _rate(rows, "wrong_target"),
        "tool_selection_accuracy": _rate(rows, "tool_selection_correct"),
        "write_verification_success_rate": _rate(write_rows, "verification_success"),
        "average_llm_calls": (sum(r.llm_calls for r in rows) / len(rows)) if rows else 0.0,
        "average_tool_calls": (sum(r.tool_calls for r in rows) / len(rows)) if rows else 0.0,
        "average_latency_ms": (sum(r.latency_ms for r in rows) / len(rows)) if rows else 0.0,
    }
