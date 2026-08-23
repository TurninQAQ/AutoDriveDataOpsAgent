"""Small offline evaluation runner over an EventStore snapshot."""
from __future__ import annotations

from .metrics import EvaluationLabels, EvaluationMetrics, evaluate_audit


def evaluate_event_store(event_store, *, labels: EvaluationLabels | None = None) -> EvaluationMetrics:
    return evaluate_audit(event_store.all(), labels=labels)
