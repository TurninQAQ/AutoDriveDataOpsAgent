"""Deterministic target relations for Runtime completion evidence."""

from __future__ import annotations

import re

from .goals import SubmitTask


_GENERATED_TASK_SUFFIX = re.compile(r"^\d{8}_\d{6}$")


def target_matches_for_goal(
    required: str,
    actual: str,
    requirement_kind: str,
    goal: object | None = None,
) -> bool:
    """Match evidence to a goal without weakening ordinary identity checks.

    ``submit_task`` accepts a validated prefix and the platform deterministically
    materializes ``<prefix>_<YYYYMMDD>_<HHMMSS>``.  Only its two post-write
    verification requirements may use that derived identity relation.  All
    other goals and evidence kinds retain exact target equality.
    """

    if required == actual:
        return True
    if not isinstance(goal, SubmitTask):
        return False
    if requirement_kind not in {"ACTION_VERIFIED", "OPERATIONAL_GOAL_VERIFIED"}:
        return False
    marker = f"{required}_"
    if not actual.startswith(marker):
        return False
    return bool(_GENERATED_TASK_SUFFIX.fullmatch(actual[len(marker) :]))
