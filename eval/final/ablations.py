from __future__ import annotations

ABLATIONS = {
    "no_goal_verification": "Evaluation-only: Action Verification success is used as the final success signal.",
    "no_evidence_provenance": "Evaluation-only: target provenance conflict is ignored.",
    "no_atomic_authorization": "Evaluation-only: legacy count-then-create race simulation for concurrency cases.",
}


def list_ablations() -> dict[str, str]:
    return dict(ABLATIONS)
