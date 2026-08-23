"""Evaluation adapters kept outside Runtime authority."""
from __future__ import annotations

from typing import Mapping

from .metrics import goal_state_macro_f1


def evaluate_goal_states(expected: Mapping[str, str], predicted: Mapping[str, str]) -> float | None:
    return goal_state_macro_f1(expected, predicted)
